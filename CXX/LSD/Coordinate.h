//
// Created by cyx on 2021/10/13.
//

#ifndef LSD_COORDINATE_H
#define LSD_COORDINATE_H


#include <iostream>


template <class T>
struct Coordinate {
    T row;
    T col;

    bool operator== (const Coordinate<T>& rhs) const {
        return (row == rhs.row && col == rhs.col);
    }

    bool operator< (const Coordinate<T>& rhs) const {
        if (row < rhs.row) {
            return true;
        }
        if (row > rhs.row) {
            return false;
        }
        return col < rhs.col;
    }

    friend std::ostream &operator<<(std::ostream &stream, const Coordinate<T> &p){
        stream << "("<< p.col << "," << p.row << ")";
        return stream;
    }
    Coordinate<T>() {
        row = 0; col = 0;
    };
    Coordinate<T>(double setRow, double setCol) {
        row = setRow; col = setCol;
    };
    Coordinate<T> operator-(const Coordinate<T>& b)
    {
        Coordinate<T> temp;
        temp.row = row - b.row;
        temp.col = col - b.col;
        return temp;
    }
    Coordinate<T> operator+(const Coordinate<T>& b)
    {
        Coordinate<T> temp;
        temp.row = row + b.row;
        temp.col = col + b.col;
        return temp;
    }
};




#endif //LSD_COORDINATE_H
